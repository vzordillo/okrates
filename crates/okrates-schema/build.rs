use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn resolve_git_dirs(dot_git: &Path) -> Option<(PathBuf, PathBuf)> {
    if dot_git.is_dir() {
        return Some((dot_git.to_path_buf(), dot_git.to_path_buf()));
    }
    let raw = fs::read_to_string(dot_git).ok()?;
    let gitdir_rel = raw.trim().strip_prefix("gitdir:")?.trim();
    let git_dir = dot_git.parent()?.join(gitdir_rel);
    let common_dir = fs::read_to_string(git_dir.join("commondir"))
        .ok()
        .map(|rel| git_dir.join(rel.trim()))
        .unwrap_or_else(|| git_dir.clone());
    Some((git_dir, common_dir))
}

fn watch_if_exists(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OKRATES_GIT_SHA");
    let dot_git = Path::new("../../.git");
    if dot_git.is_file() {
        watch_if_exists(dot_git);
    }
    if let Some((git_dir, common_dir)) = resolve_git_dirs(dot_git) {
        let head_path = git_dir.join("HEAD");
        watch_if_exists(&head_path);
        watch_if_exists(&git_dir.join("logs/HEAD"));
        watch_if_exists(&common_dir.join("packed-refs"));
        if let Ok(head) = fs::read_to_string(&head_path)
            && let Some(ref_path) = head.trim().strip_prefix("ref: ")
        {
            watch_if_exists(&common_dir.join(ref_path));
        }
    }
    let sha = env::var("OKRATES_GIT_SHA")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            Command::new("git")
                .args(["rev-parse", "--short=12", "HEAD"])
                .output()
                .ok()
                .filter(|out| out.status.success())
                .and_then(|out| String::from_utf8(out.stdout).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=OKRATES_GIT_SHA={sha}");
}
