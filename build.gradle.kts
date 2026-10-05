import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinVersion

plugins {
    id("org.jetbrains.kotlin.jvm") version "2.3.21" apply false
}

// ml-core configures itself (it is embedded into the plugins as a subtree); the rest of the modules are configured here.
configure(subprojects.filter { it.name != "ml-core" }) {
    apply(plugin = "org.jetbrains.kotlin.jvm")
    group = "io.github.completionml"
    version = "0.1.0"

    extensions.configure<org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension> {
        jvmToolchain(21)
        compilerOptions {
            // Same as the IntelliJ plugins that embed ml-core: stdlib comes from the platform (Kotlin 2.3.x).
            apiVersion.set(KotlinVersion.KOTLIN_2_3)
            languageVersion.set(KotlinVersion.KOTLIN_2_3)
            jvmTarget.set(JvmTarget.JVM_21)
            allWarningsAsErrors.set(false)
        }
    }

    dependencies {
        "testImplementation"(kotlin("test"))
        "testImplementation"("org.junit.jupiter:junit-jupiter:5.11.4")
        "testRuntimeOnly"("org.junit.platform:junit-platform-launcher")
    }

    tasks.withType<Test>().configureEach {
        useJUnitPlatform()
        maxHeapSize = "2g"
        testLogging { events("failed"); showStackTraces = true; exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL }
    }
}
