plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.testjjml"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.testjjml"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        externalNativeBuild {
            cmake {
                arguments.add("-DGGML_BACKEND_DL=OFF")
                arguments.add("-DLLAMA_BUILD_COMMON=ON")
                arguments.add("-DLLAMA_BUILD_TOOLS=ON")
                arguments.add("-DLLAMA_CURL=OFF")
//                arguments.add("-DJAVA_HOME=/usr/lib/jvm/java-17-openj9-amd64")
                arguments.add("-DCMAKE_BUILD_TYPE=Release")
            }
        }
        ndk {
            abiFilters += listOf("x86_64", "arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    externalNativeBuild {
        cmake {
            path = file("../../../CMakeLists.txt")
         }
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
}