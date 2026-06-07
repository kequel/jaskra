// swift-tools-version: 5.9

// =====================================================================
//  CI-ONLY MANIFEST
//  This Package.swift is tailored for GitHub Actions so the test target
//  can `@testable import AppModule` and produce coverage + test results.
//  It is NOT the Swift Playgrounds manifest — the Playgrounds project keeps
//  its own auto-generated manifest (with the .iOSApplication product).
//  When mirroring code from Playgrounds to git, copy the .swift source
//  files only — DO NOT overwrite this file.
// =====================================================================

import PackageDescription

let package = Package(
    name: "JaskraApp",
    platforms: [
        .iOS("18.1")
    ],
    products: [
        .library(name: "JaskraApp", targets: ["AppModule"])
    ],
    targets: [
        .target(
            name: "AppModule",
            path: ".",
            // Tests live in their own target; GlaucomaApp.swift is the @main
            // app entry (not needed in a library and not under test).
            exclude: ["Tests", "GlaucomaApp.swift"],
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals")
            ]
        ),
        .testTarget(
            name: "AppModuleTests",
            dependencies: ["AppModule"],
            path: "Tests"
        )
    ]
)
