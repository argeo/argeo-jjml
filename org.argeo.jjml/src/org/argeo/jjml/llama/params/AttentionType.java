package org.argeo.jjml.llama.params;

import java.util.function.IntSupplier;

/**
 * Attention type.
 * 
 * @see llama.h - enum llama_attention_type
 */
public enum AttentionType implements IntSupplier {
	LLAMA_ATTENTION_TYPE_UNSPECIFIED(-1), //
	LLAMA_ATTENTION_TYPE_CAUSAL(0), //
	LLAMA_ATTENTION_TYPE_NON_CAUSAL(1), //
	;

	private int code;

	private AttentionType(int code) {
		this.code = code;
	}

	@Override
	public int getAsInt() {
		return code;
	}

	public static AttentionType byCode(int code) throws IllegalArgumentException {
		for (AttentionType type : values())
			if (type.code == code)
				return type;
		throw new IllegalArgumentException("Unkown code : " + code);
	}
}
