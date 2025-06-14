package org.argeo.jjml.ggml;

import static java.lang.System.Logger.Level.INFO;
import static java.lang.System.Logger.Level.WARNING;

import java.io.File;
import java.lang.System.Logger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.argeo.jjml.internal.OsUtils;

/** A registered GGML backend. */
public class GgmlBackend {
	private final static Logger logger = System.getLogger(GgmlBackend.class.getName());

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

	private static native long doLoadBackend(byte[] backendPath);

	private static native void doLoadAllBackends(byte[] basePath);

	public static void loadAllBackends() {
		List<Path> basePaths = new ArrayList<>();
		// java.library.path
		String javaLibraryPath = System.getProperty("java.library.path");
		if (javaLibraryPath != null && !"".equals(javaLibraryPath.trim())) {
			String[] paths = javaLibraryPath.split(File.pathSeparator);
			for (String p : paths)
				basePaths.add(Paths.get(p));
		}

		// "standard" deployment paths
		if (basePaths.isEmpty())
			basePaths.add(Paths.get("/usr/libexec/x86_64-linux-gnu/ggml"));

		// load
		for (Path basePath : basePaths) {
			if (Files.exists(basePath)) {
				// loadBackends(basePath);
				doLoadAllBackends(OsUtils.filePathToNative(basePath));
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
				if (backendName.name().equals(backend.getName())) {
					logger.log(WARNING, backendName.name() + " already loaded from " + backend.getPath());
					System.err.println(backendName.name() + " already loaded from " + backend.getPath());
					continue backendNames;
				}
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
				long pointer = doLoadBackend(OsUtils.filePathToNative(basePath));
				if (pointer > 0) {
					// TODO log it
					GgmlBackend backend = new GgmlBackend(pointer, backendName.name(), backendPath);
					loadedBackends.add(backend);
					logger.log(INFO, "Loaded backend " + backendName.name() + " from " + backend.getPath());
					System.out.println("Loaded backend " + backendName.name() + " from " + backend.getPath());
				}
			}
		}
	}
}
