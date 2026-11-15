package org.argeo.jjml.llm.util.models;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A lightweight, single-class recursive descent JSON parser. Parses standard
 * JSON into Java Maps, Lists, Strings, Numbers, and Booleans.
 */
class SimpleJsonParser {

	private final String src;
	private int cursor = 0;
	private int line = 1;
	private int column = 1;

	public SimpleJsonParser(String json) {
		this.src = json != null ? json : "";
	}

	public static Object parse(String json) {
		return new SimpleJsonParser(json).parseValue();
	}

	private Object parseValue() {
		skipWhitespace();
		if (cursor >= src.length()) {
			throw error("Unexpected end of JSON input");
		}

		char c = src.charAt(cursor);
		if (c == '{')
			return parseObject();
		if (c == '[')
			return parseArray();
		if (c == '"')
			return parseString();
		if (c == 't' || c == 'f')
			return parseBoolean();
		if (c == 'n')
			return parseNull();
		if (c == '-' || Character.isDigit(c))
			return parseNumber();

		throw error("Unexpected character '" + c + "'");
	}

	private Map<String, Object> parseObject() {
		Map<String, Object> map = new HashMap<>();
		consume('{');
		skipWhitespace();

		if (peek() == '}') {
			consume('}');
			return map;
		}

		while (true) {
			skipWhitespace();
			if (peek() != '"') {
				throw error("Expected double-quoted string key in object");
			}
			String key = parseString();

			skipWhitespace();
			consume(':');

			Object value = parseValue();
			map.put(key, value);

			skipWhitespace();
			char next = peek();
			if (next == '}') {
				consume('}');
				break;
			} else if (next == ',') {
				consume(',');
			} else {
				throw error("Expected ',' or '}' inside object structure");
			}
		}
		return map;
	}

	private List<Object> parseArray() {
		List<Object> list = new ArrayList<>();
		consume('[');
		skipWhitespace();

		if (peek() == ']') {
			consume(']');
			return list;
		}

		while (true) {
			list.add(parseValue());
			skipWhitespace();
			char next = peek();
			if (next == ']') {
				consume(']');
				break;
			} else if (next == ',') {
				consume(',');
			} else {
				throw error("Expected ',' or ']' inside array structure");
			}
		}
		return list;
	}

	private String parseString() {
		consume('"');
		StringBuilder sb = new StringBuilder();
		while (cursor < src.length()) {
			char c = src.charAt(cursor);
			advanceCursor();

			if (c == '"') {
				return sb.toString();
			} else if (c == '\\') {
				if (cursor >= src.length())
					throw error("Unterminated escape sequence");
				char escape = src.charAt(cursor);
				advanceCursor();
				switch (escape) {
				case '"':
					sb.append('"');
					break;
				case '\\':
					sb.append('\\');
					break;
				case '/':
					sb.append('/');
					break;
				case 'b':
					sb.append('\b');
					break;
				case 'f':
					sb.append('\f');
					break;
				case 'n':
					sb.append('\n');
					break;
				case 'r':
					sb.append('\r');
					break;
				case 't':
					sb.append('\t');
					break;
				default:
					throw error("Unknown escape character: \\" + escape);
				}
			} else {
				sb.append(c);
			}
		}
		throw error("Unterminated string literal");
	}

	private Number parseNumber() {
		int start = cursor;
		if (src.charAt(cursor) == '-')
			advanceCursor();

		while (cursor < src.length()
				&& (Character.isDigit(src.charAt(cursor)) || "eE.-+".indexOf(src.charAt(cursor)) >= 0)) {
			advanceCursor();
		}

		String numStr = src.substring(start, cursor);
		try {
			if (numStr.contains(".") || numStr.contains("e") || numStr.contains("E")) {
				return Double.parseDouble(numStr);
			} else {
				return Long.parseLong(numStr);
			}
		} catch (NumberFormatException nfe) {
			throw error("Invalid numeric format: " + numStr);
		}
	}

	private Boolean parseBoolean() {
		if (src.startsWith("true", cursor)) {
			for (int i = 0; i < 4; i++)
				advanceCursor();
			return Boolean.TRUE;
		} else if (src.startsWith("false", cursor)) {
			for (int i = 0; i < 5; i++)
				advanceCursor();
			return Boolean.FALSE;
		}
		throw error("Invalid boolean literal sequence");
	}

	private Object parseNull() {
		if (src.startsWith("null", cursor)) {
			for (int i = 0; i < 4; i++)
				advanceCursor();
			return null;
		}
		throw error("Invalid token sequence (expected 'null')");
	}

	private void skipWhitespace() {
		while (cursor < src.length() && " \t\n\r".indexOf(src.charAt(cursor)) >= 0) {
			advanceCursor();
		}
	}

	private char peek() {
		if (cursor >= src.length())
			throw error("Unexpected EOF");
		return src.charAt(cursor);
	}

	private void consume(char expected) {
		if (cursor >= src.length() || src.charAt(cursor) != expected) {
			throw error("Expected character '" + expected + "'");
		}
		advanceCursor();
	}

	private void advanceCursor() {
		if (cursor >= src.length())
			return;
		char current = src.charAt(cursor);
		cursor++;
		if (current == '\n') {
			line++;
			column = 1;
		} else {
			column++;
		}
	}

	private JsonParseException error(String message) {
		return new JsonParseException(message, this.line, this.column);
	}

	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		System.out.println("--- Test 1: Simple Map ---");
		String s1 = "{\"id\":101,\"title\":\"Product A\",\"ok\":true}";
		Map<String, Object> m1 = (Map<String, Object>) parse(s1);
		System.out.println("Result: " + m1);

		System.out.println("\n--- Test 2: Nested Complex Map ---");
		String s2 = "{\"user\":\"Jane\",\"info\":{\"active\":false,\"val\":9.5},\"list\":[\"a\",\"b\"]}";
		Map<String, Object> m2 = (Map<String, Object>) parse(s2);
		System.out.println("User name: " + m2.get("user"));
		Map<String, Object> sub = (Map<String, Object>) m2.get("info");
		System.out.println("Nested active flag: " + sub.get("active"));

		System.out.println("\n--- Test 3: Root Array ---");
		String s3 = "[42,\"text\",false,null]";
		List<Object> l3 = (List<Object>) parse(s3);
		System.out.println("Array size: " + l3.size());

		System.out.println("\n--- Test 4: Error Handling Tracing ---");
		String bad = "{\n  \"item\": \"Pen\"\n  \"qty\": 5\n}";
		try {
			parse(bad);
		} catch (JsonParseException e) {
			System.out.println("Caught: " + e.getMessage());
		}
	}

	public static class JsonParseException extends RuntimeException {
		private static final long serialVersionUID = -4657772363409285058L;

		public JsonParseException(String message, int line, int column) {
			super(String.format("Error at line %d, column %d: %s", line, column, message));
		}
	}

}
