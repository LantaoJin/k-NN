use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let knn_jni_root = PathBuf::from(&manifest_dir)
        .parent() // rust-port
        .and_then(|p| p.parent()) // jni
        .expect("Could not resolve jni root directory")
        .to_path_buf();

    // Link against faiss and nmslib.
    // Use dynamic linking by default; set KNN_JNI_STATIC=1 for static linking.
    let link_kind = if env::var("KNN_JNI_STATIC").unwrap_or_default() == "1" {
        "static"
    } else {
        "dylib"
    };

    // Only add static library paths when NOT in stub-link mode.
    // In stub-link mode, we only use the dynamic stubs from build/release.
    if env::var("KNN_JNI_STUB_LINK").is_err() {
        // C++ shim (knn_shim.cpp compiled to libknn_shim.a)
        let shim_dir = PathBuf::from(&manifest_dir).join("csrc");
        if shim_dir.join("libknn_shim.a").exists() {
            println!("cargo:rustc-link-search=native={}", shim_dir.display());
            println!("cargo:rustc-link-lib=static=knn_shim");
        }

        let faiss_c_api_dir = knn_jni_root.join("build").join("faiss_c_api");
        if faiss_c_api_dir.exists() {
            println!("cargo:rustc-link-search=native={}", faiss_c_api_dir.display());
        }

        let faiss_static_dir = knn_jni_root.join("build").join("external").join("faiss").join("faiss");
        if faiss_static_dir.exists() {
            println!("cargo:rustc-link-search=native={}", faiss_static_dir.display());
        }
    }

    // Determine the library search path.
    // Prefer the KNN_JNI_LIB_DIR environment variable; fall back to the
    // default build/release directory relative to the k-NN jni folder.
    let lib_dir = env::var("KNN_JNI_LIB_DIR").unwrap_or_else(|_| {
        knn_jni_root
            .join("build")
            .join("release")
            .to_string_lossy()
            .into_owned()
    });
    println!("cargo:rustc-link-search=native={}", lib_dir);

    println!("cargo:rustc-link-lib={}=faiss", link_kind);
    println!("cargo:rustc-link-lib={}=nmslib", link_kind);

    // Also link against OpenMP and standard C++ library which faiss/nmslib
    // typically require.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=gomp");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if cfg!(target_os = "macos") {
        // OpenMP and libc++ only needed when linking against real faiss/nmslib.
        // Skip if KNN_JNI_STUB_LINK is set (for Phase C testing with stub libs).
        if std::env::var("KNN_JNI_STUB_LINK").is_err() {
            println!("cargo:rustc-link-search=native=/opt/homebrew/opt/libomp/lib");
            println!("cargo:rustc-link-lib=dylib=omp");
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    }

    // Re-run if the library directory contents change.
    println!("cargo:rerun-if-env-changed=KNN_JNI_LIB_DIR");
    println!("cargo:rerun-if-env-changed=KNN_JNI_STATIC");
    println!("cargo:rerun-if-env-changed=KNN_JNI_STUB_LINK");
    println!("cargo:rerun-if-changed=build.rs");

    // Optional: generate Rust bindings via bindgen if the feature or env var
    // is set. This requires the `bindgen` crate as a build-dependency.
    #[cfg(feature = "generate_bindings")]
    {
        let bindings = bindgen::Builder::default()
            .header("wrapper.h")
            .clang_arg(format!("-I{}", lib_dir))
            .generate()
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Could not write bindings");
    }
}
