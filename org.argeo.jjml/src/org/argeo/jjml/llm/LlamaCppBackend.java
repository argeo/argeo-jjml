package org.argeo.jjml.llm;

import org.argeo.jjml.ggml.params.NumaStrategy;
import org.argeo.jjml.llm.params.ContextParams;
import org.argeo.jjml.llm.params.ModelParams;

/**
 * Wrapper to the llama.cpp backend. Only static methods as this is a singleton
 * on the native side.
 * <p>
 * All other objects in this package are guaranteed to check that the backend
 * has been initialized, and {@link #destroy()} is called by JVM shutdown. For
 * simple use cases, there should therefore be no need to use this class.
 * </p>
 */
public class LlamaCppBackend {

	static {
		LlamaCppNative.ensureLibrariesLoaded();
		// Make sure that we will destroy the backend properly on JVM shutdown.
		Runtime.getRuntime().addShutdownHook(new Thread(() -> destroy(), "Destroy llama.cpp backend"));
	}

	/*
	 * NATIVE
	 */
	static native void doNumaInit(int numaStrategyCode);

	static native void doDestroy();

	static native ModelParams newModelParams();

	static native ContextParams newContextParams();

	public static native boolean supportsMmap();

	public static native boolean supportsMlock();

	public static native boolean supportsGpuOffload();

	/*
	 * LIFECYCLE
	 */

	/**
	 * Initialize NUMA strategy.
	 */
	public static void numaInit(NumaStrategy numaStrategy) {
		if (numaStrategy != null)
			doNumaInit(numaStrategy.getAsInt());
	}

	/**
	 * Destroy the backend. Note that the JNI library won't be unloaded until the
	 * related classloader has been garbage-collected.
	 */
	public static void destroy() {
		doDestroy();
	}

	/** singleton */
	private LlamaCppBackend() {
	}
}
