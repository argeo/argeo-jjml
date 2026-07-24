package org.argeo.jjml.llm.params;

/** Names of the supported model parameters. */
public enum ModelParam {
	n_gpu_layers, //
	vocab_only, //
	/** @deprecated ignored */
	@Deprecated
	use_mmap, //
	/** @deprecated ignored */
	@Deprecated
	use_mlock, //
	;

	/**
	 * System property to set model parameters, such as the default number of layers
	 * offloaded to GPU.
	 */
	final static String SYSTEM_PROPERTY_MODEL_PARAM_PREFIX = "jjml.llm.model.";

	/** As a system property used to override default value. */
	public String asSystemProperty() {
		return SYSTEM_PROPERTY_MODEL_PARAM_PREFIX + name();
	}
}
