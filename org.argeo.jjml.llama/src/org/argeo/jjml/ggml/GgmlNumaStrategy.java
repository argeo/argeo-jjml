package org.argeo.jjml.ggml;

/**
 * GGML NUMA strategy enumeration.
 * 
 * @see ggml.h - enum ggml_numa_strategy
 */
public enum GgmlNumaStrategy {
	GGML_NUMA_STRATEGY_DISABLED(0), //
	GGML_NUMA_STRATEGY_DISTRIBUTE(1), //
	GGML_NUMA_STRATEGY_ISOLATE(2), //
	GGML_NUMA_STRATEGY_NUMACTL(3), //
	GGML_NUMA_STRATEGY_MIRROR(4), //
	;

	private int code;

	private GgmlNumaStrategy(int code) {
		this.code = code;
	}

	public int getCode() {
		return code;
	}

	public static GgmlNumaStrategy byCode(int code) throws IllegalArgumentException {
		for (GgmlNumaStrategy type : values())
			if (type.code == code)
				return type;
		throw new IllegalArgumentException("Unkown pooling type code : " + code);
	}

}
