package org.argeo.jjml.internal;

import java.nio.charset.Charset;
import java.nio.file.Path;

/** OS- or locale- dependent utilities, mostly around encoding. */
public class OsUtils {
	final static Charset OS_CHARSET;

	static {
		// Note: IBM OpenJ9 seems to support this property as well
		String encodingSysProp = System.getProperty("sun.jnu.encoding");
		if (encodingSysProp != null)
			OS_CHARSET = Charset.forName(encodingSysProp);
		else
			throw new IllegalStateException("Cannot find default OS encoding");
	}

	public static byte[] filePathToNative(Path path) {
		return filePathToNative(path.toString());
	}

	public static byte[] filePathToNative(String path) {
		return path.getBytes(OS_CHARSET);
	}

	public static void main(String[] args) {
		System.out.println(OS_CHARSET);
	}
}
