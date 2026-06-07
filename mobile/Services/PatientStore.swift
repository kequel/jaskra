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
        patients.sort { $0.fullName.localizedCaseInsensitiveCompare($1.fullName) == .orderedAscending }
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

    /// Persist a completed analysis for a patient, storing the overlay image on disk.
    @discardableResult
    func addRecord(for patient: Patient, result: GlaucomaResult, image: UIImage?) -> AnalysisRecord {
        var filename: String?
        if let image, let data = image.jpegData(compressionQuality: 0.9) {
            let name = "\(UUID().uuidString).jpg"
            try? data.write(to: imagesDir.appendingPathComponent(name), options: .atomic)
            filename = name
        }

        let record = AnalysisRecord(
            patientId: patient.id,
            hasGlaucoma: result.hasGlaucoma,
            confidence: result.confidence,
            cupToDiscRatio: result.cupToDiscRatio,
            imageFilename: filename
        )
        records.append(record)
        saveRecords()
        return record
    }

    func deleteRecord(_ record: AnalysisRecord) {
        deleteImageFile(record.imageFilename)
        records.removeAll { $0.id == record.id }
        saveRecords()
    }

    func image(for record: AnalysisRecord) -> UIImage? {
        guard let filename = record.imageFilename else { return nil }
        let url = imagesDir.appendingPathComponent(filename)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return UIImage(data: data)
    }

    // MARK: - Private

    private func deleteImageFile(_ filename: String?) {
        guard let filename else { return }
        try? FileManager.default.removeItem(at: imagesDir.appendingPathComponent(filename))
    }
}
