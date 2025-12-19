package org.argeo.jjml.mtmd;

import static java.nio.charset.StandardCharsets.UTF_8;

public class MtmdBackend {
	private final static String DEFAULT_MARKER;

	static {
		MtmdNative.ensureLibrariesLoaded();

		DEFAULT_MARKER = new String(doGetDefaultMarker(), UTF_8);
	}

	private static native byte[] doGetDefaultMarker();

	public static String getDefaultMarker() {
		return DEFAULT_MARKER;
	}

	/** singleton */
	private MtmdBackend() {
	}
}
