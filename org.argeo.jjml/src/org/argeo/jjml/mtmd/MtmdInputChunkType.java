package org.argeo.jjml.mtmd;

import java.util.function.IntSupplier;

public enum MtmdInputChunkType implements IntSupplier {
	TEXT(0), //
	IMAGE(1), //
	AUDIO(2), //
	;

	private int code;

	private MtmdInputChunkType(int code) {
		this.code = code;
	}

	@Override
	public int getAsInt() {
		return code;
	}

}
