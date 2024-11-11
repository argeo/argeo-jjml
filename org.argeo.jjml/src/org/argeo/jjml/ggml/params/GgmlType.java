package org.argeo.jjml.ggml.params;

import java.util.function.IntSupplier;

/**
 * GGML type.
 * 
 * @see ggml.h - enum ggml_type
 */
public enum GgmlType implements IntSupplier {
	GGML_TYPE_F32(0), //
	GGML_TYPE_F16(1), //
	GGML_TYPE_Q4_0(2), //
	GGML_TYPE_Q4_1(3), //
	// GGML_TYPE_Q4_2(4), // support has been removed
	// GGML_TYPE_Q4_3(5), // support has been removed
	GGML_TYPE_Q5_0(6), //
	GGML_TYPE_Q5_1(7), //
	GGML_TYPE_Q8_0(8), //
	GGML_TYPE_Q8_1(9), //
	GGML_TYPE_Q2_K(10), //
	GGML_TYPE_Q3_K(11), //
	GGML_TYPE_Q4_K(12), //
	GGML_TYPE_Q5_K(13), //
	GGML_TYPE_Q6_K(14), //
	GGML_TYPE_Q8_K(15), //
	GGML_TYPE_IQ2_XXS(16), //
	GGML_TYPE_IQ2_XS(17), //
	GGML_TYPE_IQ3_XXS(18), //
	GGML_TYPE_IQ1_S(19), //
	GGML_TYPE_IQ4_NL(20), //
	GGML_TYPE_IQ3_S(21), //
	GGML_TYPE_IQ2_S(22), //
	GGML_TYPE_IQ4_XS(23), //
	GGML_TYPE_I8(24), //
	GGML_TYPE_I16(25), //
	GGML_TYPE_I32(26), //
	GGML_TYPE_I64(27), //
	GGML_TYPE_F64(28), //
	GGML_TYPE_IQ1_M(29), //
	GGML_TYPE_BF16(30), //
	GGML_TYPE_Q4_0_4_4(31), //
	GGML_TYPE_Q4_0_4_8(32), //
	GGML_TYPE_Q4_0_8_8(33), //
	GGML_TYPE_TQ1_0(34), //
	GGML_TYPE_TQ2_0(35), //
	;

	private int code;

	private GgmlType(int code) {
		this.code = code;
	}

	@Override
	public int getAsInt() {
		return code;
	}

	public static GgmlType byCode(int code) throws IllegalArgumentException {
		for (GgmlType type : values())
			if (type.code == code)
				return type;
		throw new IllegalArgumentException("Unkown code : " + code);
	}

}
