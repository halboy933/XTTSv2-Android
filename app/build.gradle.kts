plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.dorama.xtts"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.dorama.xtts"
        minSdk = 31
        targetSdk = 36
        versionCode = 14
        versionName = "0.3.10-stage3k"
    }

   signingConfigs {
    create("release") {
        storeFile = file(
            System.getenv("XTTS_KEYSTORE_PATH")
                ?: "${System.getProperty("java.io.tmpdir")}/xtts-release.jks"
        )
        storePassword = System.getenv("XTTS_KEYSTORE_PASSWORD") ?: ""
        keyAlias = System.getenv("XTTS_KEY_ALIAS") ?: "xtts"
        keyPassword = System.getenv("XTTS_KEY_PASSWORD") ?: ""
    }
}
    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.13.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
}