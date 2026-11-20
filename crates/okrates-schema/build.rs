use std::collections::BTreeSet;
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

fn watch_repo_worktree(repo_root: &Path) {
    let Ok(output) = Command::new("git")
        .current_dir(repo_root)
        .args(["-c", "core.quotePath=false", "ls-files", "-z"])
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }
    let mut top_dirs = BTreeSet::new();
    for raw in output.stdout.split(|&b| b == 0) {
        if raw.is_empty() {
            continue;
        }
        let Ok(rel_str) = std::str::from_utf8(raw) else {
            continue;
        };
        let rel = Path::new(rel_str);
        watch_if_exists(&repo_root.join(rel));
        // Watch tracked top-level directories (`crates`, `python`, ...) without watching `..` (which contains `target/` and `.git/`).
        let mut comps = rel.components();
        if let Some(first) = comps.next()
            && comps.next().is_some()
        {
            top_dirs.insert(PathBuf::from(first.as_os_str()));
        }
    }
    for dir in top_dirs {
        watch_if_exists(&repo_root.join(dir));
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OKRATES_GIT_SHA");
    println!("cargo:rerun-if-env-changed=OKRATES_GIT_DIRTY");
    let repo_root = Path::new("../..");
    let dirty_override = non_empty_env("OKRATES_GIT_DIRTY");
    let dot_git = repo_root.join(".git");
    if dot_git.is_file() {
        watch_if_exists(&dot_git);
    }
    let git_dirs = resolve_git_dirs(&dot_git);
    if let Some((git_dir, common_dir)) = &git_dirs {
        let head_path = git_dir.join("HEAD");
        watch_if_exists(&head_path);
        watch_if_exists(&git_dir.join("index"));
        watch_if_exists(&git_dir.join("logs/HEAD"));
        watch_if_exists(&common_dir.join("packed-refs"));
        if let Ok(head) = fs::read_to_string(&head_path)
            && let Some(ref_path) = head.trim().strip_prefix("ref: ")
        {
            watch_if_exists(&common_dir.join(ref_path));
        }
        if dirty_override.is_none() {
            watch_repo_worktree(repo_root);
        }
    }
    let sha = non_empty_env("OKRATES_GIT_SHA")
        .or_else(|| {
            Command::new("git")
                .current_dir(repo_root)
                .args(["rev-parse", "--short=12", "HEAD"])
                .output()
                .ok()
                .filter(|out| out.status.success())
                .and_then(|out| String::from_utf8(out.stdout).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = dirty_override
        .or_else(|| {
            git_dirs.as_ref().and_then(|_| {
                Command::new("git")
                    .current_dir(repo_root)
                    .args(["status", "--porcelain"])
                    .output()
                    .ok()
                    .filter(|out| out.status.success())
                    .and_then(|out| String::from_utf8(out.stdout).ok())
                    .map(|s| {
                        if s.trim().is_empty() {
                            "false".to_string()
                        } else {
                            "true".to_string()
                        }
                    })
            })
        })
        .unwrap_or_else(|| "unknown".to_string());
    let target = non_empty_env("TARGET").unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=OKRATES_GIT_SHA={sha}");
    println!("cargo:rustc-env=OKRATES_GIT_DIRTY={dirty}");
    println!("cargo:rustc-env=OKRATES_TARGET={target}");
}
