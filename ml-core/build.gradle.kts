// Pure Kotlin, no IntelliJ dependencies: tokens, vocabulary, n-gram LM, ranking model, feature extraction, model and shard
// formats, metrics. Embedded into IDE plugins via git subtree (docs/ADAPTER.md), so this build file is self-sufficient:
// it does not rely on the root project's configuration and must not pull third-party runtime libraries
// (class-loader conflicts, plugin size). The Kotlin plugin version comes from the host's root `plugins` block.
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinVersion

plugins {
    id("org.jetbrains.kotlin.jvm")
}

repositories {
    mavenCentral()
}

kotlin {
    compilerOptions {
        // Same as the IntelliJ plugins: the platform bundles its own Kotlin stdlib (2.3.x), don't use newer API.
        apiVersion.set(KotlinVersion.KOTLIN_2_3)
        languageVersion.set(KotlinVersion.KOTLIN_2_3)
        jvmTarget.set(JvmTarget.JVM_21)
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

dependencies {
    // `kotlin.stdlib.default.dependency = false` in the plugins: declare it explicitly, the host provides the real one at runtime
    implementation(kotlin("stdlib"))
    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    maxHeapSize = "2g"
    testLogging { events("failed"); showStackTraces = true; exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL }
}

// Prints the runtime classpath for running the benchmark main (io.github.completionml.core.nn.NnBenchKt) with plain `java`.
val printBenchClasspath by tasks.registering {
    val cp = sourceSets.test.get().runtimeClasspath
    dependsOn(tasks.named("testClasses"))
    doLast { println("CP=" + cp.asPath) }
}
