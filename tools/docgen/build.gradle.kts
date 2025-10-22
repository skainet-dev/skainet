plugins {
    kotlin("jvm")
    alias(libs.plugins.kotlinSerialization)
    application
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.2")
    implementation("org.jetbrains.kotlinx:kotlinx-cli:0.3.6")
    
    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.1")
}

application {
    mainClass.set("sk.ainet.tools.docgen.DocGenKt")
}

tasks.test {
    useJUnitPlatform()
}
