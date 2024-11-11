package org.argeo.jjml.llama.params;

import static java.lang.Boolean.parseBoolean;
import static java.lang.Integer.parseInt;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

import org.argeo.jjml.llama.LlamaCppModel;

/**
 * Initialization parameters of a model. New instance should be created by using
 * the {@link #with(Map)} methods on {@link LlamaCppModel#defaultModelParams()},
 * with the default values populated by the shared library.
 * 
 * Note: it provides record-style getters, in order to ease transition to
 * Java records in the future.
 * 
 * @see llama.h - llama_model_params
 */
public class ModelParams {
	private final int n_gpu_layers;
	private final boolean vocab_only;
	private final boolean use_mlock;

	/** Record-like full constructor */
	public ModelParams( //
			int n_gpu_layers, //
			boolean vocab_only, //
			boolean use_mlock //
	) {
		this.n_gpu_layers = n_gpu_layers;
		this.vocab_only = vocab_only;
		this.use_mlock = use_mlock;
	}

	public ModelParams with(ModelParamName key, Object value) {
		Objects.requireNonNull(key);
		Objects.requireNonNull(value);
		return with(Collections.singletonMap(key, value.toString()));
	}

	public ModelParams with(Map<ModelParamName, String> p) {
		return new ModelParams( //
				parseInt(p.getOrDefault(ModelParamName.n_gpu_layers, Integer.toString(this.n_gpu_layers))), //
				parseBoolean(p.getOrDefault(ModelParamName.vocab_only, Boolean.toString(this.vocab_only))), //
				parseBoolean(p.getOrDefault(ModelParamName.use_mlock, Boolean.toString(this.use_mlock))) //
		);
	}

	public int n_gpu_layers() {
		return n_gpu_layers;
	}

	public boolean vocab_only() {
		return vocab_only;
	}

	public boolean use_mlock() {
		return use_mlock;
	}

//	/** Ensure that components and enum are perfectly in line. */
//	private static boolean assertParamNames() {
//		RecordComponent[] components = ModelParams.class.getRecordComponents();
//		ModelParamName[] names = ModelParamName.values();
//		assert components.length == names.length;
//		for (int i = 0; i < names.length; i++) {
//			assert components[i].getName().equals(names[i].name());
//		}
//		return true;
//	}

}
