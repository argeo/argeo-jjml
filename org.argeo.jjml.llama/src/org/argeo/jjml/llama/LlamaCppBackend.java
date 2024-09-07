package org.argeo.jjml.llama;

/** Wrapper to the llama.cpp backend. There can be only one instance. */
public class LlamaCppBackend {
	private static LlamaCppBackend instance;

	native void doInit();

	native void doDestroy();

	public void init() {
		if (instance != null)
			throw new IllegalStateException("llama.cpp backend is already initialized.");
		System.loadLibrary("org_argeo_jjml_llama");
		doInit();
		instance = this;
	}

	public void destroy() {
		doDestroy();
		instance = null;
		// TODO classloading so that library can be unloaded by garbage collection
	}
}
