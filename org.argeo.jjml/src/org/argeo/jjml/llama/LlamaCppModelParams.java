package org.argeo.jjml.llama;

import static java.lang.Boolean.parseBoolean;
import static java.lang.Integer.parseInt;

import java.lang.reflect.RecordComponent;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Initialization parameters of a model. New instance should be created with
 * {@link #defaultModelParams()}, so that the shared library can be used to
 * populate the default values.
 * 
 * @see llama.h - llama_model_params
 */
public record LlamaCppModelParams(int n_gpu_layers, boolean vocab_only, boolean use_mlock) {

	public static enum ParamName {
		n_gpu_layers, //
		vocab_only, //
		use_mlock, //
		;
	}

	final static LlamaCppModelParams DEFAULT;

	static {
		assert assertParamNames();
		DEFAULT = defaultModelParams();
	}

//	LlamaCppModelParams(Map<ParamName, String> p) {
//		this( //
//				parseInt(p.getOrDefault(ParamName.n_gpu_layers, Integer.toString(DEFAULT.n_gpu_layers))), //
//				parseBoolean(p.getOrDefault(ParamName.vocab_only, Boolean.toString(DEFAULT.vocab_only))), //
//				parseBoolean(p.getOrDefault(ParamName.use_mlock, Boolean.toString(DEFAULT.use_mlock))) //
//		);
//	}

	public LlamaCppModelParams with(ParamName name, Object value) {
		Objects.requireNonNull(name);
		Objects.requireNonNull(value);
		return with(Collections.singletonMap(name, value.toString()));
	}

	public LlamaCppModelParams with(Map<ParamName, String> p) {
		return new LlamaCppModelParams( //
				parseInt(p.getOrDefault(ParamName.n_gpu_layers, Integer.toString(this.n_gpu_layers))), //
				parseBoolean(p.getOrDefault(ParamName.vocab_only, Boolean.toString(this.vocab_only))), //
				parseBoolean(p.getOrDefault(ParamName.use_mlock, Boolean.toString(this.use_mlock))) //
		);
	}

	/**
	 * The default model parameters.
	 * 
	 * @see llama.h - llama_model_default_params()
	 */
	static LlamaCppModelParams defaultModelParams() {
		LlamaCppNative.ensureLibrariesLoaded();
		return null;// LlamaCppNative.newModelParams();
	}

	/** Ensure that components and enum are in line. */
	private static boolean assertParamNames() {
		RecordComponent[] components = LlamaCppModelParams.class.getRecordComponents();
		ParamName[] names = ParamName.values();
		assert components.length == names.length;
		for (int i = 0; i < components.length; i++) {
			assert components[i].getName().equals(names[i].name());
		}
		return true;

	}

//	public LinkedHashMap<String, Object> toMap() {
//		LinkedHashMap<String, Object> res = new LinkedHashMap<>();
//		for (ParamName paramName : ParamName.values()) {
//
//		}
//
//		Class<? extends Record> clss = LlamaCppModelParams.class;
//		RecordComponent[] components = clss.getRecordComponents();
//		for (var comp : components) {
//			try {
//				Field field = clss.getDeclaredField(comp.getName());
//				field.setAccessible(true);
//				Object value = field.get(this);
//				res.put(comp.getName(), value);
//			} catch (NoSuchFieldException | IllegalArgumentException | IllegalAccessException e) {
//			}
//		}
//		return res;
//	}
}
