package org.argeo.jjml.llama;

/**
 * Initialization parameters of a model. New instance should be created with
 * {@link #defaultModelParams()}, so that the shared library can be used to
 * populate the default values.
 * 
 * @see llama.h - llama_model_params
 */
public class LlamaCppModelParams {
	int gpuLayersCount;

	boolean vocabOnly;

	boolean useMlock;

	LlamaCppModelParams() {
	}

	/**
	 * The default model parameters.
	 * 
	 * @see llama.h - llama_model_default_params()
	 */
	public static LlamaCppModelParams defaultModelParams() {
		LlamaCppNative.ensureLibrariesLoaded();
		return LlamaCppNative.newModelParams();
	}

	public int getGpuLayersCount() {
		return gpuLayersCount;
	}

	public void setGpuLayersCount(int gpuLayersCount) {
		this.gpuLayersCount = gpuLayersCount;
	}

	public boolean isVocabOnly() {
		return vocabOnly;
	}

	public void setVocabOnly(boolean vocabOnly) {
		this.vocabOnly = vocabOnly;
	}

	public boolean isUseMlock() {
		return useMlock;
	}

	public void setUseMlock(boolean useMlock) {
		this.useMlock = useMlock;
	}

}
