package org.argeo.jjml.ggml.params;

import java.util.function.IntSupplier;

/**
 * GGML NUMA strategy enumeration.
 * 
 * @see ggml.h - enum ggml_numa_strategy
 */
public enum NumaStrategy implements IntSupplier {
	GGML_NUMA_STRATEGY_DISABLED(0), //
	GGML_NUMA_STRATEGY_DISTRIBUTE(1), //
	GGML_NUMA_STRATEGY_ISOLATE(2), //
	GGML_NUMA_STRATEGY_NUMACTL(3), //
	GGML_NUMA_STRATEGY_MIRROR(4), //
	;

	private int code;

	private NumaStrategy(int code) {
		this.code = code;
	}

	@Override
	public int getAsInt() {
		return code;
	}

	public static NumaStrategy byCode(int code) throws IllegalArgumentException {
		for (NumaStrategy type : values())
			if (type.code == code)
				return type;
		throw new IllegalArgumentException("Unkown code : " + code);
	}

}
