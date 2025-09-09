package org.argeo.jjml.mtmd;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import javax.imageio.ImageIO;

public class MtmdImageBitmap extends MtmdBitmap {

	MtmdImageBitmap(ByteBuffer rgb, int width, int height) {
		super(doInit(rgb, width, height));
	}

	private static native long doInit(ByteBuffer rgb, int width, int height);

	@Override
	public MtmdInputChunkType getType() {
		return MtmdInputChunkType.IMAGE;
	}

	static MtmdImageBitmap load(InputStream in) throws IOException {
		BufferedImage img = ImageIO.read(in);
		int width = img.getWidth();
		int height = img.getHeight();
		// FIXME make sure the native buffer is properly freed
		ByteBuffer buf = ByteBuffer.allocateDirect(width * height * 3);
		buf.order(ByteOrder.nativeOrder());
		// TODO optimize with direct access to the image buffers?
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int argb = img.getRGB(x, y);
				int red = (argb >> 16) & 0x000000FF;
				int green = (argb >> 8) & 0x000000FF;
				int blue = argb & 0x000000FF;
				buf.put((byte) red);
				buf.put((byte) green);
				buf.put((byte) blue);
			}
		}

		return new MtmdImageBitmap(buf, width, height);
	}
}
