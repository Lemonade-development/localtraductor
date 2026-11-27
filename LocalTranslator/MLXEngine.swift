import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
// HuggingFace y Tokenizers no se usan directamente en este archivo, pero la
// expansión del macro `#huggingFaceLoadModel` genera código que referencia
// `HubClient` y `Tokenizers`: sin estos imports el proyecto no compila.
import HuggingFace
import Tokenizers
import OSLog

/// Motor de traducción real. Usa MLX para correr un LLM cuantizado
/// en 4-bit sobre Apple Silicon.
///
/// La primera vez que se carga, descarga los pesos del modelo desde
/// Hugging Face (~1 GB para Hy-MT2-1.8B) y los guarda en caché local
/// dentro del contenedor de la app. Las siguientes veces, carga
/// directamente desde disco.
///
/// Es un `actor` para que toda la inferencia esté serializada y nunca
/// haya dos generaciones a la vez sobre el mismo modelo.
actor MLXEngine: TranslationEngine {

    // MARK: - Configuración

    /// Identificador del modelo en el repo `mlx-community` de Hugging Face.
    /// Para cambiar de modelo, basta con pasar otro ID al init.
    private let modelID: String

    /// Sesión de chat ya montada sobre el modelo cargado.
    /// La creamos una sola vez en `loadModel()` y la reutilizamos.
    private var session: ChatSession?

    /// Canal de logs visible en Console.app filtrando por subsystem
    /// `LocalTranslator` / category `MLXEngine`. `nonisolated` para poder
    /// usarlo desde los closures `@Sendable` del progreso de descarga sin
    /// salto al actor.
    private nonisolated let log = Logger(subsystem: "LocalTranslator", category: "MLXEngine")

    init(modelID: String = ModelStorage.modelID) {
        self.modelID = modelID
    }

    /// Suelta la sesión (y con ella el modelo) para que MLX libere los
    /// ~1 GB de RAM. Se usa al eliminar el modelo desde Configuración.
    func unload() {
        session = nil
        log.info("Modelo descargado de memoria")
    }

    // MARK: - Carga del modelo

    func loadModel(progressHandler: (@Sendable (Double) -> Void)? = nil) async throws {
        log.info("Iniciando carga del modelo: \(self.modelID, privacy: .public)")
        let configuration = ModelConfiguration(id: modelID)

        // 1) Descarga (la primera vez) + carga en memoria.
        //    Reportamos el progreso al logger y, si nos pasaron handler,
        //    también se lo entregamos para que la UI lo refleje.
        let model: ModelContext
        do {
            model = try await loadMLXModelContext(for: configuration) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                let done = ByteCountFormatter.string(
                    fromByteCount: progress.completedUnitCount,
                    countStyle: .file
                )
                let total = ByteCountFormatter.string(
                    fromByteCount: progress.totalUnitCount,
                    countStyle: .file
                )
                self.log.debug("Descarga \(pct)% (\(done, privacy: .public) / \(total, privacy: .public))")
                progressHandler?(progress.fractionCompleted)
            }
        } catch {
            log.error("Fallo cargando modelo: \(error.localizedDescription, privacy: .public)")
            throw error
        }
        log.info("Modelo cargado en memoria")

        // 2) Parámetros de generación recomendados por Tencent para
        //    Hy-MT2 1.8B / 7B (ver README del repo Hy-MT2).
        let parameters = GenerateParameters(
            maxTokens: 4096,
            temperature: 0.7,
            topP: 0.6,
            topK: 20,
            repetitionPenalty: 1.05
        )

        // 3) Sin system prompt: Hy-MT2 está entrenado sin él y todas las
        //    instrucciones van en el mensaje de usuario (ver `translate`).
        self.session = ChatSession(
            model,
            generateParameters: parameters
        )
    }

    // MARK: - Traducción

    func translate(_ text: String,
                   from source: Language,
                   to target: Language,
                   tone: TranslationTone) async throws -> AsyncThrowingStream<String, Error> {

        guard let session else { throw EngineError.modelNotLoaded }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return AsyncThrowingStream { $0.finish() }
        }

        try Task.checkCancellation()

        // Cada traducción debe ser independiente: limpiamos el historial
        // para que el modelo no use traducciones previas como contexto.
        await session.clear()

        // Seguimos la plantilla oficial de Hy-MT2 ("Translate the following
        // text into X. Note that you should only output the translated
        // result…") y le añadimos el tono y las reglas para conservar
        // markdown y los placeholders `⟦Cn⟧` que la app inyecta para
        // proteger código y URLs (ver `MarkdownCodePreserver`).
        let toneClause = tone.instruction.map { " " + $0 } ?? ""
        let prompt = """
        Translate the following text from \(source.englishName) \
        into \(target.englishName).\(toneClause) \
        Preserve markdown structure and paragraph breaks. \
        Copy placeholder tokens like ⟦C0⟧, ⟦C1⟧ exactly as-is. \
        Note that you should only output the translated result \
        without any additional explanation:

        \(trimmed)
        """

        // Devolvemos directamente el stream del ChatSession: la UI irá
        // concatenando los chunks conforme el modelo genera tokens.
        return session.streamResponse(to: prompt)
    }
}

// MARK: - Carga del modelo (función libre)

/// Helper a nivel de archivo para cargar el modelo fuera del actor. Hace
/// falta porque dentro del actor `loadModel` queda resuelto a nuestro método
/// de instancia y los macros de MLXHuggingFace no pueden expandirse ahí.
///
/// No usamos el macro `#huggingFaceLoadModel`, que va contra el `HubClient`
/// por defecto: ese cliente guarda los pesos en `Library/Caches`, que macOS
/// puede purgar (ver `ModelStorage`). Montamos el downloader a mano con un
/// `HubClient` cuyo caché apunta a Application Support.
private func loadMLXModelContext(
    for configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> ModelContext {
    let hub = HubClient(cache: HubCache(cacheDirectory: ModelStorage.hubCacheDirectory))
    return try await MLXLMCommon.loadModel(
        from: #hubDownloader(hub),
        using: #huggingFaceTokenizerLoader(),
        configuration: configuration,
        progressHandler: progressHandler
    )
}

// MARK: - Errores propios

enum EngineError: LocalizedError {
    case modelNotLoaded
    /// El traductor del sistema soporta el par pero los paquetes de idioma
    /// aún no están descargados. El ViewModel lo captura para lanzar el
    /// flujo de descarga (ver `LanguageDownloadWindowController`).
    case languagesNotInstalled(source: Language, target: Language)
    /// El traductor del sistema no soporta este par de idiomas.
    case languagePairUnsupported(source: Language, target: Language)

    var errorDescription: String? {
        // `String(localized:)` para que el mensaje siga el idioma del
        // sistema; estos textos acaban dentro del "Error: %@" de la UI.
        switch self {
        case .modelNotLoaded:
            return String(localized: "El modelo aún no está cargado.")
        case .languagesNotInstalled:
            return String(localized: "Faltan idiomas de traducción por descargar.")
        case let .languagePairUnsupported(source, target):
            return String(localized: "El traductor del sistema no soporta \(source.englishName) → \(target.englishName).")
        }
    }
}
