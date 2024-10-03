package org.argeo.jjml.llama;

import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

import org.argeo.jjml.ggml.GgmlNumaStrategy;

/** Wrapper to the llama.cpp backend. There can be only one instance. */
public class LlamaCppBackend {
	/**
	 * System property to explicitly specify the path of the GGML shared library to
	 * use.
	 */
	public final static String SYSTEM_PROPERTY_LIBPATH_GGML = "org.argeo.jjml.libpath.ggml";
	/**
	 * System property to explicitly specify the path of the llama.cpp shared
	 * library to use.
	 */
	public final static String SYSTEM_PROPERTY_LIBPATH_LLAMACPP = "org.argeo.jjml.libpath.llamacpp";
	/**
	 * System property to explicitly specify the path of the JNI shared library to
	 * use.
	 */
	public final static String SYSTEM_PROPERTY_LIBPATH_JJML_LLAMA = "org.argeo.jjml.libpath.jjml.llama";

	private final static String JJML_LAMA_LIBRARY_NAME = "org_argeo_jjml_llama";

	private final static Logger logger = System.getLogger(LlamaCppBackend.class.getName());

	private static boolean initialized = false;
	private static boolean librariesLoaded = false;

	private static Path ggmlLibraryPath;
	private static Path llamaLibraryPath;
	private static Path jjmlLlamaLibraryPath;

	/** Cannot be instantiated. */
	private LlamaCppBackend() {
	}

	/*
	 * NATIVE
	 */
	static native void doInit();

	static native void doNumaInit(int numaStrategyCode);

	static native void doDestroy();

	/*
	 * LIFECYCLE
	 */
	/**
	 * Initialize the backend.
	 * 
	 * @see llama.h - llama_backend_init()
	 */
	public static void init() {
		init(null);
	}

	/**
	 * Initialize the backend with a NUMA strategy.
	 * 
	 * @see llama.h - llama_backend_init()
	 * @see llama.h - llama_numa_init()
	 */
	public static void init(GgmlNumaStrategy numaStrategy) {
		// TODO register the stack where the library was already loaded
		if (initialized)
			throw new IllegalStateException("llama.cpp backend is already initialized.");

		if (!librariesLoaded) {
			// Initialization fails if explicitly found libraries are not found, since it
			// assumed that it will be used in a development or debugging context, which can
			// be messy.
			Optional.ofNullable(System.getProperty(SYSTEM_PROPERTY_LIBPATH_GGML)).ifPresent((path) -> {
				ggmlLibraryPath = Paths.get(path);
				if (!Files.exists(ggmlLibraryPath))
					throw new IllegalArgumentException(
							SYSTEM_PROPERTY_LIBPATH_GGML + " " + ggmlLibraryPath + " does not exist");
			});
			Optional.ofNullable(System.getProperty(SYSTEM_PROPERTY_LIBPATH_LLAMACPP)).ifPresent((path) -> {
				llamaLibraryPath = Paths.get(path);
				if (!Files.exists(llamaLibraryPath))
					throw new IllegalArgumentException(
							SYSTEM_PROPERTY_LIBPATH_LLAMACPP + " " + llamaLibraryPath + " does not exist");
			});
			Optional.ofNullable(System.getProperty(SYSTEM_PROPERTY_LIBPATH_JJML_LLAMA)).ifPresent((path) -> {
				jjmlLlamaLibraryPath = Paths.get(path);
				if (!Files.exists(jjmlLlamaLibraryPath))
					throw new IllegalArgumentException(
							SYSTEM_PROPERTY_LIBPATH_JJML_LLAMA + " " + jjmlLlamaLibraryPath + " does not exist");
			});

			if (ggmlLibraryPath != null) {
				System.load(ggmlLibraryPath.toAbsolutePath().toString());
				logger.log(Level.WARNING, "GGML library loaded from " + ggmlLibraryPath);
			}
			if (llamaLibraryPath != null) {
				System.load(llamaLibraryPath.toAbsolutePath().toString());
				logger.log(Level.WARNING, "llama.cpp library loaded from " + llamaLibraryPath);
			}
			if (jjmlLlamaLibraryPath != null) {
				System.load(jjmlLlamaLibraryPath.toAbsolutePath().toString());
			}

			// default behavior
			System.loadLibrary(JJML_LAMA_LIBRARY_NAME);
			librariesLoaded = true;
		}

		doInit();
		initialized = true;
		if (numaStrategy != null)
			doNumaInit(numaStrategy.getCode());
	}

	/** Ensure that the backend has been initialized. */
	public static void ensureInitialized() {
		if (!initialized)
			init();
	}

	/**
	 * Destroy the backend. Note that the JNI library won't be unloaded until the
	 * related classloader has been garbage-collected.
	 * 
	 * @see llama.h - llama_backend_free()
	 */
	public static void destroy() {
		if (!initialized)
			return; // ignore silently
		doDestroy();
		initialized = false;
		// TODO classloading so that libraries can be unloaded by garbage collection
	}

	public static void setJjmlLlamaLibraryPath(Path jjmlLlamaLibraryPath) {
		if (librariesLoaded)
			throw new IllegalStateException("Shared libraries are already loaded.");
		LlamaCppBackend.jjmlLlamaLibraryPath = jjmlLlamaLibraryPath;
	}

	public static void setLlamaLibraryPath(Path llamaLibraryPath) {
		if (librariesLoaded)
			throw new IllegalStateException("Shared libraries are already loaded.");
		LlamaCppBackend.llamaLibraryPath = llamaLibraryPath;
	}

	public static void setGgmlLibraryPath(Path ggmlLibraryPath) {
		if (librariesLoaded)
			throw new IllegalStateException("Shared libraries are already loaded.");
		LlamaCppBackend.ggmlLibraryPath = ggmlLibraryPath;
	}
}
