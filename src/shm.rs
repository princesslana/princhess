//! Read-only data shared between processes, on huge pages.

use std::cmp::Reverse;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::ptr::NonNull;
use std::time::SystemTime;

use bytemuck::Pod;

use crate::mem::HUGE_PAGE_SIZE;

// Enough for the engines in a test not to remove each other's files
const FILES_KEPT: usize = 9;

/// Shares `embedded` through a file named by `key`, which must identify the contents. Falls back
/// to `embedded` itself.
pub fn share<T: Pod>(key: &str, embedded: &'static T) -> &'static T {
    match map(key, bytemuck::bytes_of(embedded)) {
        // SAFETY: The mapping is huge page aligned, at least as large as a T, never unmapped, and
        // holds a copy of `embedded`. Any bytes of the right size are a valid Pod.
        Some(shared) => unsafe { shared.cast::<T>().as_ref() },
        None => embedded,
    }
}

fn map(key: &str, bytes: &[u8]) -> Option<NonNull<u8>> {
    // Make sure the key is a valid MD5 hash, not build.rs's "unknown" or "none"
    if key.len() != 32 || !key.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }

    let dir = sys::private_dir()?;
    let len = bytes.len().next_multiple_of(HUGE_PAGE_SIZE);
    let path = dir.join(format!("{key}-{}.bin", bytes.len()));

    let file = match File::open(&path) {
        Ok(file) if file.metadata().ok()?.len() == len as u64 => file,
        _ => {
            let file = create(&path, bytes, len).ok()?;
            remove_old(&dir, &path);
            file
        }
    };

    sys::map_file(&file, len)
}

// Renamed into place, so a file at `path` is always complete
fn create(path: &Path, bytes: &[u8], len: usize) -> io::Result<File> {
    let tmp = path.with_extension(format!("{}.tmp", process::id()));
    let mut file = OpenOptions::new().write(true).create_new(true).open(&tmp)?;

    let written = file
        .write_all(bytes)
        .and_then(|()| file.set_len(len as u64))
        .and_then(|()| fs::rename(&tmp, path));

    if let Err(e) = written {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }

    File::open(path)
}

// Processes mapping a removed file keep it until they exit
fn remove_old(dir: &Path, current: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    let mut old: Vec<(SystemTime, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path != current)
        .filter_map(|path| Some((fs::metadata(&path).ok()?.modified().ok()?, path)))
        .collect();

    old.sort_by_key(|&(modified, _)| Reverse(modified));

    for (_, path) in old.into_iter().skip(FILES_KEPT) {
        let _ = fs::remove_file(path);
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use std::ffi::CStr;
    use std::fs::{self, DirBuilder, File};
    use std::io;
    use std::mem::MaybeUninit;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    use std::path::{Path, PathBuf};
    use std::ptr::{self, NonNull};

    use crate::mem::HUGE_PAGE_SIZE;

    const ROOT: &CStr = c"/dev/shm";

    // libc only defines it for glibc. The kernel's value is the same on every architecture:
    // https://github.com/torvalds/linux/blob/v6.1/include/uapi/asm-generic/mman-common.h#L80
    const MADV_COLLAPSE: libc::c_int = 25;

    // Read-only and shared, huge page aligned, and never unmapped
    pub fn map_file(file: &File, len: usize) -> Option<NonNull<u8>> {
        // SAFETY: The reservation is only used to find a huge page aligned range, which the file
        // mapping then replaces with MAP_FIXED. madvise only changes paging hints.
        unsafe {
            let reserved = libc::mmap(
                ptr::null_mut(),
                len + HUGE_PAGE_SIZE,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            );
            if reserved == libc::MAP_FAILED {
                return None;
            }

            let aligned = (reserved as usize).next_multiple_of(HUGE_PAGE_SIZE);
            let mapped = libc::mmap(
                aligned as *mut libc::c_void,
                len,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_FIXED,
                file.as_raw_fd(),
                0,
            );
            if mapped == libc::MAP_FAILED {
                libc::munmap(reserved, len + HUGE_PAGE_SIZE);
                return None;
            }

            // MADV_HUGEPAGE is ignored for shared memory unless shmem_enabled allows it
            libc::madvise(mapped, len, libc::MADV_HUGEPAGE);
            libc::madvise(mapped, len, MADV_COLLAPSE);

            NonNull::new(mapped.cast())
        }
    }

    // On tmpfs, so the files are shared memory, and usable by this user alone
    pub fn private_dir() -> Option<PathBuf> {
        if !is_tmpfs(ROOT) {
            return None;
        }

        let uid = current_uid();
        let dir = Path::new(ROOT.to_str().ok()?).join(format!("princhess-{uid}"));

        match DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(_) => return None,
        }

        let meta = fs::symlink_metadata(&dir).ok()?;
        (meta.is_dir() && meta.uid() == uid && meta.mode() & 0o7777 == 0o700).then_some(dir)
    }

    fn is_tmpfs(path: &CStr) -> bool {
        let mut stat = MaybeUninit::<libc::statfs>::zeroed();

        // SAFETY: statfs only writes to the struct, and it's read only after statfs succeeds
        unsafe {
            libc::statfs(path.as_ptr(), stat.as_mut_ptr()) == 0
                && stat.assume_init_ref().f_type == libc::TMPFS_MAGIC
        }
    }

    fn current_uid() -> u32 {
        // SAFETY: getuid has no preconditions and can't fail
        unsafe { libc::getuid() }
    }
}

#[cfg(not(target_os = "linux"))]
mod sys {
    use std::fs::File;
    use std::path::PathBuf;
    use std::ptr::NonNull;

    pub fn map_file(_file: &File, _len: usize) -> Option<NonNull<u8>> {
        None
    }

    pub fn private_dir() -> Option<PathBuf> {
        None
    }
}
