package org.argeo.jjml.llama.params;

import java.util.function.IntSupplier;

/**
 * Pooling type.
 * 
 * @see llama.h - enum llama_pooling_type
 */
public enum PoolingType implements IntSupplier {
	LLAMA_POOLING_TYPE_UNSPECIFIED(-1), //
	LLAMA_POOLING_TYPE_NONE(0), //
	LLAMA_POOLING_TYPE_MEAN(1), //
	LLAMA_POOLING_TYPE_CLS(2), //
	LLAMA_POOLING_TYPE_LAST(3), //
	;

	private int code;

	private PoolingType(int code) {
		this.code = code;
	}

	@Override
	public int getAsInt() {
		return code;
	}

	public static PoolingType byCode(int code) throws IllegalArgumentException {
		for (PoolingType type : values())
			if (type.code == code)
				return type;
		throw new IllegalArgumentException("Unkown pooling type code : " + code);
	}
}
