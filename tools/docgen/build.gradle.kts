plugins {
    kotlin("jvm")
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.asciidoctorJvm)
    application
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.cli)

    implementation(libs.asciidoctorj.core)

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.1")
}

application {
    mainClass.set("sk.ainet.tools.docgen.DocGenKt")
}

tasks.named<JavaExec>("run") {
    args = listOf(
        "--input", "${project.rootDir}/skainet-lang/skainet-lang-core/build/generated/ksp/metadata/commonMain/resources/operators.json",
        "--output", "${project.rootDir}/docs/modules/operators/_generated_"
    )
}

tasks.test {
    useJUnitPlatform()
}
