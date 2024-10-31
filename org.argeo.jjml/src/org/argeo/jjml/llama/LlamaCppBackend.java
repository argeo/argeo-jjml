package org.argeo.jjml.llama;

import java.lang.System.Logger;

import org.argeo.jjml.ggml.GgmlNumaStrategy;

/** Wrapper to the llama.cpp backend. There can be only one instance. */
public class LlamaCppBackend {

	private final static Logger logger = System.getLogger(LlamaCppBackend.class.getName());

	/** Default executor to use for asynchronous IO operations. */
//	private static ExecutorService ioExecutor = null;

	private static boolean initialized = false;

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
		if (initialized)
			throw new IllegalStateException("llama.cpp backend is already initialized.");

		// Make sure that we will destroy the backend properly on JVM shutdown.
		Runtime.getRuntime().addShutdownHook(new Thread(() -> destroy(), "Destroy llama.cpp backend"));

		LlamaCppNative.ensureLibrariesLoaded();

		doInit();
		initialized = true;

		if (numaStrategy != null)
			doNumaInit(numaStrategy.getAsInt());

//		ioExecutor = Executors.newCachedThreadPool();
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

		// complete blocking operations
//		if (ioExecutor != null)
//			ioExecutor.shutdown();

		doDestroy();
		initialized = false;
		// TODO classloading so that libraries can be unloaded by garbage collection
	}

	/** Executor for IO operations (file system, network). */
//	static Executor getIoExecutor() {
//		return ioExecutor;
//	}
}
