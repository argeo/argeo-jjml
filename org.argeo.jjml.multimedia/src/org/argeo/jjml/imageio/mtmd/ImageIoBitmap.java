package org.argeo.jjml.imageio.mtmd;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import javax.imageio.ImageIO;

import org.argeo.jjml.mtmd.MtmdImageBitmap;

public class ImageIoBitmap extends MtmdImageBitmap {

	ImageIoBitmap(ByteBuffer rgb, int width, int height) {
		super(rgb, width, height);
	}

	public static MtmdImageBitmap load(InputStream in) throws IOException {
		BufferedImage img = ImageIO.read(in);
		int width = img.getWidth();
		int height = img.getHeight();

//		BufferedImage testImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

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

//				int rgb = red;
//				rgb = (rgb << 8) + green;
//				rgb = (rgb << 8) + blue;
//				testImg.setRGB(x, y, rgb);
			}
		}

//		ImageIO.write(testImg, "bmp", new File("test.bmp"));

		return new ImageIoBitmap(buf, width, height);
	}

}
