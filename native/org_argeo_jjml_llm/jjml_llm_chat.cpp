#include <string>
#include <vector>
#include <cstring>
#include <cassert>

#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_LlamaCppNativeChatFormatter.h" // IWYU pragma: keep

/*
 * CHAT
 */
JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llm_LlamaCppNativeChatFormatter_doFormatChatMessages(
		JNIEnv *env, jclass, jobjectArray roles, jobjectArray contents,
		jboolean addAssistantTokens, jbyteArray chatTemplateStr) {
	const jsize messages_size = env->GetArrayLength(roles);
	assert(env->GetArrayLength(contents) == messages_size);

	std::vector<llama_chat_message> chat_messages;

	try {
		std::vector<std::string> role_strs;
		std::vector<std::string> content_strs;
		role_strs.reserve(messages_size);
		content_strs.reserve(messages_size);
		int alloc_size = 0;

		for (int i = 0; i < messages_size; i++) {
			role_strs.push_back(argeo::jni::to_string(env, roles, i));
			content_strs.push_back(argeo::jni::to_string(env, contents, i));
			// using the same factor as in common.cpp
			alloc_size += (role_strs[i].length() + content_strs[i].length())
					* 1.25;
		}

		chat_messages.reserve(messages_size);
		for (int i = 0; i < messages_size; i++) {
			chat_messages.push_back(
					{ role_strs[i].c_str(), content_strs[i].c_str() });
		}

		std::string u8_chat_template;
		if (chatTemplateStr != nullptr)
			u8_chat_template = argeo::jni::to_string(env, chatTemplateStr);

		std::vector<char> buf(alloc_size);
		int32_t resLength = llama_chat_apply_template(
				chatTemplateStr != nullptr ? u8_chat_template.c_str() : nullptr,
				chat_messages.data(), chat_messages.size(), addAssistantTokens,
				buf.data(), buf.size());

		// error: chat template is not supported
		if (resLength < 0) {
			if (chatTemplateStr != nullptr)
				throw std::runtime_error("Custom template is not supported");
			else
				throw std::runtime_error("Built-in template is not supported");
		}

		// if it turns out that our buffer is too small, we resize it
		if ((size_t) resLength > buf.size()) {
			buf.resize(resLength);
			resLength = llama_chat_apply_template(
					chatTemplateStr != nullptr ?
							u8_chat_template.c_str() : nullptr,
					chat_messages.data(), chat_messages.size(),
					addAssistantTokens, buf.data(), buf.size());
		}

		std::string u8_res(buf.data(), resLength);
		jbyteArray res = env->NewByteArray(u8_res.length());
		env->SetByteArrayRegion(res, 0, u8_res.length(), (jbyte*) &u8_res[0]);
		return res;
	} catch (...) {
		return argeo::jni::throw_to_java(env);
	}
}
