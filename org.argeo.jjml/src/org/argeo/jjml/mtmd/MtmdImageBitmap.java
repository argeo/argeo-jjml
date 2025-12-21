package org.argeo.jjml.mtmd;

import java.nio.ByteBuffer;

public class MtmdImageBitmap extends MtmdBitmap {
	protected MtmdImageBitmap(ByteBuffer rgb, int width, int height) {
		super(doInit(rgb, width, height));
	}

	protected MtmdImageBitmap(byte[] rgb, int offset, int width, int height) {
		super(doInitFromBytes(rgb, offset, width, height));
	}

	private static native long doInit(ByteBuffer rgb, int width, int height);

	private static native long doInitFromBytes(byte[] rgb, int offset, int width, int height);

	@Override
	public MtmdInputChunkType getType() {
		return MtmdInputChunkType.IMAGE;
	}

	@Override
	public void close() throws Exception {
		super.close();
	}

}
