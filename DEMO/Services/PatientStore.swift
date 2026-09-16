import SwiftUI
import UIKit

// =====================================================================
//  PATIENT STORE
//  Local persistence for patients and their analysis history.
//  Patients + records are saved as JSON in Application Support; processed
//  images are written as JPEG files alongside them.
//  (The backend does not model patients yet — see the backend issue doc.)
// =====================================================================

@MainActor
final class PatientStore: ObservableObject {
    @Published private(set) var patients: [Patient] = []
    @Published private(set) var records: [AnalysisRecord] = []

    private let patientsFile = "patients.json"
    private let recordsFile = "records.json"

    init() {
        load()
        if patients.isEmpty {
            seedDemoPatients()
        }
    }

    // MARK: - Storage locations

    private var baseDir: URL {
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = support.appendingPathComponent("Jaskra", isDirectory: true)
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    private var imagesDir: URL {
        let fm = FileManager.default
        let dir = baseDir.appendingPathComponent("images", isDirectory: true)
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    // MARK: - Load / save

    private func load() {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        if let data = try? Data(contentsOf: baseDir.appendingPathComponent(patientsFile)),
           let decoded = try? decoder.decode([Patient].self, from: data) {
            patients = decoded
        }
        if let data = try? Data(contentsOf: baseDir.appendingPathComponent(recordsFile)),
           let decoded = try? decoder.decode([AnalysisRecord].self, from: data) {
            records = decoded
        }
        sortPatients()
    }

    private func savePatients() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        if let data = try? encoder.encode(patients) {
            try? data.write(to: baseDir.appendingPathComponent(patientsFile), options: .atomic)
        }
    }

    private func saveRecords() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        if let data = try? encoder.encode(records) {
            try? data.write(to: baseDir.appendingPathComponent(recordsFile), options: .atomic)
        }
    }

    private func sortPatients() {
        patients.sort { lhs, rhs in
            let byLastName = lhs.lastName.localizedCaseInsensitiveCompare(rhs.lastName)
            if byLastName != .orderedSame { return byLastName == .orderedAscending }
            return lhs.firstName.localizedCaseInsensitiveCompare(rhs.firstName) == .orderedAscending
        }
    }

    // MARK: - Patient CRUD

    func addPatient(_ patient: Patient) {
        patients.append(patient)
        sortPatients()
        savePatients()
    }

    func updatePatient(_ patient: Patient) {
        guard let idx = patients.firstIndex(where: { $0.id == patient.id }) else { return }
        patients[idx] = patient
        sortPatients()
        savePatients()
    }

    func deletePatient(_ patient: Patient) {
        // Remove the patient's images, then their records, then the patient.
        for record in records where record.patientId == patient.id {
            deleteImageFile(record.imageFilename)
            deleteImageFile(record.maskFilename)
        }
        records.removeAll { $0.patientId == patient.id }
        patients.removeAll { $0.id == patient.id }
        savePatients()
        saveRecords()
    }

    func patient(by id: UUID) -> Patient? {
        patients.first { $0.id == id }
    }

    // MARK: - Records

    func records(for patientId: UUID) -> [AnalysisRecord] {
        records.filter { $0.patientId == patientId }.sorted { $0.date > $1.date }
    }

    var allRecordsByDate: [AnalysisRecord] {
        records.sorted { $0.date > $1.date }
    }

    func analysisCount(for patientId: UUID) -> Int {
        records.reduce(0) { $1.patientId == patientId ? $0 + 1 : $0 }
    }

    /// Persist a completed analysis for a patient, storing the overlay image
    /// (and, for the demo comparison screen, a standalone mask image) on disk.
    @discardableResult
    func addRecord(
        for patient: Patient,
        result: GlaucomaResult,
        image: UIImage?,
        maskImage: UIImage? = nil,
        date: Date = Date()
    ) -> AnalysisRecord {
        func store(_ image: UIImage?) -> String? {
            guard let image, let data = image.jpegData(compressionQuality: 0.9) else { return nil }
            let name = "\(UUID().uuidString).jpg"
            try? data.write(to: imagesDir.appendingPathComponent(name), options: .atomic)
            return name
        }

        let record = AnalysisRecord(
            patientId: patient.id,
            date: date,
            hasGlaucoma: result.hasGlaucoma,
            confidence: result.confidence,
            cupToDiscRatio: result.cupToDiscRatio,
            imageFilename: store(image),
            maskFilename: store(maskImage)
        )
        records.append(record)
        saveRecords()
        return record
    }

    func deleteRecord(_ record: AnalysisRecord) {
        deleteImageFile(record.imageFilename)
        deleteImageFile(record.maskFilename)
        records.removeAll { $0.id == record.id }
        saveRecords()
    }

    func image(for record: AnalysisRecord) -> UIImage? {
        loadImage(named: record.imageFilename)
    }

    /// Standalone disc/cup mask image for the demo comparison screen, if one
    /// was saved for this record.
    func maskImage(for record: AnalysisRecord) -> UIImage? {
        loadImage(named: record.maskFilename)
    }

    private func loadImage(named filename: String?) -> UIImage? {
        guard let filename else { return nil }
        let url = imagesDir.appendingPathComponent(filename)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return UIImage(data: data)
    }

    // MARK: - Demo seed data

    /// First-launch only: populates the app with real fundus photos run
    /// through our actual AI pipeline (see DemoPatientSeed.swift), so the
    /// doctor sees genuine model output instead of an empty app. One extra
    /// patient gets a second, older visit so "Porównaj analizy" has real
    /// images to show.
    private func seedDemoPatients() {
        for seed in DemoPatientSeed.all {
            let patient = Patient(
                firstName: "Pacjent",
                lastName: seed.code,
                email: "",
                avatarKind: seed.avatarKind,
                avatarTint: seed.avatarTint
            )
            addPatient(patient)
            addSeedRecord(seed, for: patient, date: Date())
        }

        let comparison = DemoPatientSeed.comparison
        let followUpPatient = Patient(
            firstName: "Pacjent",
            lastName: comparison.patientCode,
            email: "",
            avatarKind: comparison.avatarKind,
            avatarTint: comparison.avatarTint
        )
        addPatient(followUpPatient)
        let fourMonthsAgo = Calendar.current.date(byAdding: .month, value: -4, to: Date()) ?? Date()
        addSeedRecord(comparison.older, for: followUpPatient, date: fourMonthsAgo)
        addSeedRecord(comparison.newer, for: followUpPatient, date: Date())
    }

    private func addSeedRecord(_ seed: DemoSeedPatient, for patient: Patient, date: Date) {
        guard let imageData = Data(base64Encoded: seed.imageBase64),
              let image = UIImage(data: imageData) else { return }
        let maskImage = Data(base64Encoded: seed.maskBase64).flatMap { UIImage(data: $0) }

        let result = GlaucomaResult(
            hasGlaucoma: seed.hasGlaucoma,
            confidence: seed.confidence,
            cupToDiscRatio: seed.cdr,
            imageBase64: seed.imageBase64,
            maskImageBase64: seed.maskBase64
        )
        addRecord(for: patient, result: result, image: image, maskImage: maskImage, date: date)
    }

    // MARK: - Private

    private func deleteImageFile(_ filename: String?) {
        guard let filename else { return }
        try? FileManager.default.removeItem(at: imagesDir.appendingPathComponent(filename))
    }
}
