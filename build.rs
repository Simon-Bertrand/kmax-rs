fn main() {
    println!("cargo:rerun-if-env-changed=OPENBLAS_DLL_PATH");
    if let Ok(path) = std::env::var("OPENBLAS_DLL_PATH") {
        println!("cargo:rustc-link-search=native={}", path);
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}
