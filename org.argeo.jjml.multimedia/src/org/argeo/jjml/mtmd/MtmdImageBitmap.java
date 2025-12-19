package org.argeo.jjml.mtmd;

import java.nio.ByteBuffer;

public class MtmdImageBitmap extends MtmdBitmap {
	private ByteBuffer rgb;

	protected MtmdImageBitmap(ByteBuffer rgb, int width, int height) {
		super(doInit(rgb, width, height));
		this.rgb = rgb;
	}

	private static native long doInit(ByteBuffer rgb, int width, int height);

	@Override
	public MtmdInputChunkType getType() {
		return MtmdInputChunkType.IMAGE;
	}

	@Override
	public void close() throws Exception {
		super.close();
		rgb = null;
	}

}
