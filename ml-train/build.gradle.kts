// JVM CLI for offline training/evaluation. Not shipped in plugins; free to use heavier dependencies if ever needed.
plugins {
    application
}

dependencies {
    implementation(project(":ml-core"))
}

application {
    mainClass.set("io.github.completionml.train.MainKt")
    applicationDefaultJvmArgs = listOf("-Xmx6g", "-XX:+UseParallelGC")
}
