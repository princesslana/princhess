use std::alloc::{self, Layout};
use std::boxed::Box;

pub fn boxed_and_zeroed<T>() -> Box<T> {
    unsafe {
        let layout = Layout::new::<T>();
        let ptr = alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Box::from_raw(ptr.cast())
    }
}
