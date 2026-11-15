package com.example.testjjml;

import android.graphics.Bitmap;
import android.graphics.Color;

import org.argeo.jjml.mtmd.MtmdImageBitmap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class AndroidBitmap extends MtmdImageBitmap {

    private AndroidBitmap(ByteBuffer rgb, int width, int height) {
        super(rgb, width, height);
    }

    public static AndroidBitmap fromBitmap(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        ByteBuffer buf = ByteBuffer.allocateDirect(width * height * 3);
        buf.order(ByteOrder.nativeOrder());

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int pixel : pixels) {
            buf.put((byte) Color.red(pixel));
            buf.put((byte) Color.green(pixel));
            buf.put((byte) Color.blue(pixel));
        }
        buf.flip();

        return new AndroidBitmap(buf, width, height);
    }
}
