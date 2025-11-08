package org.argeo.jjml.llm.params;

import java.util.function.IntSupplier;

/**
 * Rope scaling type. (see enum <code>llama_rope_scaling_type</code>, in
 * llama.h)
 */
public enum RopeScalingType implements IntSupplier {
	LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED(-1), //
	LLAMA_ROPE_SCALING_TYPE_NONE(0), //
	LLAMA_ROPE_SCALING_TYPE_LINEAR(1), //
	LLAMA_ROPE_SCALING_TYPE_YARN(2), //
	;

	private int code;

	private RopeScalingType(int code) {
		this.code = code;
	}

	@Override
	public int getAsInt() {
		return code;
	}

	public static RopeScalingType byCode(int code) throws IllegalArgumentException {
		for (RopeScalingType type : values())
			if (type.code == code)
				return type;
		throw new IllegalArgumentException("Unkown pooling type code : " + code);
	}
}
