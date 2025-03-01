package org.argeo.jjml.llama.params;

/** Names of the supported model parameters. */
public enum ModelParam {
	n_gpu_layers, //
	vocab_only, //
	use_mmap, //
	use_mlock, //
	;

	/** System property to set the default number of layers offloaded to GPU. */
	final static String SYSTEM_PROPERTY_MODEL_PARAM_PREFIX = "jjml.llama.model.";

	/** As a system property used to override default value.*/
	public String asSystemProperty() {
		return SYSTEM_PROPERTY_MODEL_PARAM_PREFIX + name();
	}
}
