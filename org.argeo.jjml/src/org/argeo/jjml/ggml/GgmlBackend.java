package org.argeo.jjml.ggml;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/** A registered GGML backend. */
public class GgmlBackend {
	private final static String GGML_DL_PREFIX = "ggml-";
	// TODO rather rely on native-side registration
	private final static List<GgmlBackend> loadedBackends = new ArrayList<>();

	/** Pointer to ggml_backend_reg_t. */
	private final long pointer;
	private final String name;
	private final Path path;

	public GgmlBackend(long pointer, String name, Path path) {
		this.pointer = pointer;
		this.name = name;
		this.path = path;
	}

	private static native long doLoadBackend(String backendPath);

	private static native void doLoadAllBackends(String basePath);

	public static void loadAllBackends() {
		List<Path> basePaths = new ArrayList<>();
		// java.library.path
		String javaLibraryPath = System.getProperty("java.library.path");
//		if (javaLibraryPath != null && !"".equals(javaLibraryPath.trim())) {
//			String[] paths = javaLibraryPath.split(File.pathSeparator);
//			for (String p : paths)
//				basePaths.add(Paths.get(p));
//		}

		// "standard" deployment paths
		basePaths.add(Paths.get("/usr/libexec/x86_64-linux-gnu/ggml"));

		// load
		for (Path basePath : basePaths) {
			if (Files.exists(basePath)) {
				// loadBackends(basePath);
				doLoadAllBackends(basePath.toString());
			}
		}
	}

	public String getName() {
		return name;
	}

	public Path getPath() {
		return path;
	}

	/** Load backends available in this directory. */
	public static void loadBackends(Path basePath) {
		// TODO recurse? could be useful for local builds

		backendNames: for (StandardBackend backendName : StandardBackend.values()) {
			// skip backends whose names are already loaded
			for (GgmlBackend backend : loadedBackends) {
				if (backendName.name().equals(backend.getName()))
					continue backendNames;
			}

			String dllName;
			if (File.separatorChar == '\\')
				dllName = GGML_DL_PREFIX + backendName.name() + ".dll";
			else {
				// FIXME deal with MacOS
				dllName = "lib" + GGML_DL_PREFIX + backendName.name() + ".so";
			}
			Path backendPath = basePath.resolve(dllName);
			if (Files.exists(backendPath)) {
				long pointer = doLoadBackend(backendPath.toString());
				if (pointer > 0) {
					// TODO log it
					GgmlBackend backend = new GgmlBackend(pointer, backendName.name(), backendPath);
					loadedBackends.add(backend);
				}
			}
		}
	}
}
