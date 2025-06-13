use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{channel, Receiver, SendError, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

enum WorkerMessage<'scope> {
    RunTask(Box<dyn FnOnce() + Send + 'scope>),
    Terminate,
}

struct ScopedSender<'scope> {
    sender: Sender<WorkerMessage<'static>>,
    _marker: PhantomData<&'scope ()>,
}

impl<'scope> ScopedSender<'scope> {
    unsafe fn send_task<F>(&self, f: F) -> Result<(), SendError<WorkerMessage<'static>>>
    where
        F: FnOnce() + Send + 'scope,
    {
        let task: Box<dyn FnOnce() + Send + 'scope> = Box::new(f);
        let task_static: Box<dyn FnOnce() + Send + 'static> = std::mem::transmute(task);
        self.sender.send(WorkerMessage::RunTask(task_static))
    }
}

pub struct Scope<'scope> {
    sender: ScopedSender<'scope>,
    active_tasks: Arc<Mutex<usize>>,
    task_completed_cvar: Arc<Condvar>,
}

impl<'scope> Scope<'scope> {
    /// Spawns a task that can borrow from the current scope.
    /// The task is guaranteed to complete before the `ThreadPool::scope` call returns.
    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        *self.active_tasks.lock().unwrap() += 1;

        // SAFETY: The `ThreadPool::scope` method guarantees that this `Scope`
        // and all tasks spawned from it will complete before the 'scope lifetime ends.
        let send_result = unsafe { self.sender.send_task(f) };

        // If sending failed (e.g., channel disconnected during shutdown),
        // decrement the counter to prevent deadlock in Scope::drop.
        if send_result.is_err() {
            *self.active_tasks.lock().unwrap() -= 1;
        }
    }
}

impl Drop for Scope<'_> {
    fn drop(&mut self) {
        // Wait for all tasks spawned within this scope to complete.
        // This is the crucial part that ensures safety for the transmuted lifetimes.
        let mut active = self.active_tasks.lock().unwrap();
        while *active > 0 {
            active = self.task_completed_cvar.wait(active).unwrap();
        }
    }
}

// Helper struct to ensure active_tasks is decremented even on panic
struct TaskCompletionGuard {
    active_tasks: Arc<Mutex<usize>>,
    task_completed_cvar: Arc<Condvar>,
}

impl Drop for TaskCompletionGuard {
    fn drop(&mut self) {
        *self.active_tasks.lock().unwrap() -= 1;
        self.task_completed_cvar.notify_one();
    }
}

struct Worker {
    _id: usize,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn new(
        id: usize,
        receiver: Arc<Mutex<Receiver<WorkerMessage<'static>>>>,
        active_tasks: Arc<Mutex<usize>>,
        task_completed_cvar: Arc<Condvar>,
    ) -> Self {
        let thread = thread::spawn(move || loop {
            let message = receiver.lock().unwrap().recv().unwrap();

            match message {
                WorkerMessage::RunTask(task) => {
                    let _guard = TaskCompletionGuard {
                        active_tasks: Arc::clone(&active_tasks),
                        task_completed_cvar: Arc::clone(&task_completed_cvar),
                    };

                    // Catch panics from the user-provided task to ensure the counter is decremented.
                    let result = catch_unwind(AssertUnwindSafe(task));

                    if let Err(e) = result {
                        std::panic::resume_unwind(e);
                    }
                }
                WorkerMessage::Terminate => {
                    break;
                }
            }
        });

        Worker {
            _id: id,
            thread: Some(thread),
        }
    }
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Sender<WorkerMessage<'static>>,
    active_tasks: Arc<Mutex<usize>>,
    task_completed_cvar: Arc<Condvar>,
    scope_guard: Mutex<()>,
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0, "ThreadPool must have at least one thread.");

        let (sender, receiver) = channel();
        let receiver = Arc::new(Mutex::new(receiver));

        let active_tasks = Arc::new(Mutex::new(0));
        let task_completed_cvar = Arc::new(Condvar::new());

        let mut workers = Vec::with_capacity(num_threads);
        for id in 0..num_threads {
            workers.push(Worker::new(
                id,
                Arc::clone(&receiver),
                Arc::clone(&active_tasks),
                Arc::clone(&task_completed_cvar),
            ));
        }

        ThreadPool {
            workers,
            sender,
            active_tasks,
            task_completed_cvar,
            scope_guard: Mutex::new(()),
        }
    }

    /// Executes a closure within a scope, allowing tasks to borrow from that scope.
    /// Blocks until all spawned tasks within the scope are complete.
    pub fn scope<'scope, F>(&'scope self, f: F)
    where
        F: FnOnce(&Scope<'scope>),
    {
        let _guard = self
            .scope_guard
            .try_lock()
            .expect("ThreadPool::scope called concurrently, which is not allowed.");

        // The 'static lifetime on `WorkerMessage` is a lie, but it's made safe
        // by the blocking `wait` in the `Scope`'s `Drop` implementation.
        let scoped_sender = ScopedSender {
            sender: self.sender.clone(),
            _marker: PhantomData,
        };

        let scope_obj = Scope {
            sender: scoped_sender,
            active_tasks: Arc::clone(&self.active_tasks),
            task_completed_cvar: Arc::clone(&self.task_completed_cvar),
        };

        f(&scope_obj);
    }

    pub fn current_num_threads(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            self.sender.send(WorkerMessage::Terminate).ok();
        }
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}
