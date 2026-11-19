import type { QualityReport } from "./quality";
import type { StepId } from "./suggestions";

export interface CaptureProject {
  id: string;
  name: string;
  createdAt: string;
  steps: Array<{ id: StepId; marked: boolean }>;
}

export interface CapturePhoto {
  id: string;
  projectId: string;
  createdAt: string;
  source: "camera" | "file" | "simulated";
  width: number;
  height: number;
  blob: Blob;
  quality: QualityReport | null;
}

export interface CaptureStore {
  listProjects(): Promise<CaptureProject[]>;
  createProject(name: string): Promise<CaptureProject>;
  deleteProject(id: string): Promise<void>;
  setStep(projectId: string, step: StepId, marked: boolean): Promise<void>;
  listPhotos(projectId: string): Promise<CapturePhoto[]>;
  addPhoto(photo: CapturePhoto): Promise<void>;
  deletePhoto(id: string): Promise<void>;
  saveQuality(id: string, quality: QualityReport): Promise<void>;
  dump(): Promise<{ projects: CaptureProject[]; photos: CapturePhoto[] }>;
  load(snapshot: { projects: CaptureProject[]; photos: CapturePhoto[] }): Promise<void>;
}

const defaultSteps = (): CaptureProject["steps"] =>
  (["front", "right", "back", "left", "top", "detail"] as StepId[]).map((id) => ({ id, marked: false }));

export function memoryStore(): CaptureStore {
  let projects: CaptureProject[] = [];
  let photos: CapturePhoto[] = [];
  return {
    async listProjects() { return projects.map(cloneProject); },
    async createProject(name) {
      const project = { id: crypto.randomUUID(), name: name.trim() || "未命名拍摄", createdAt: new Date().toISOString(), steps: defaultSteps() };
      projects = [project, ...projects];
      return cloneProject(project);
    },
    async deleteProject(id) {
      projects = projects.filter((project) => project.id !== id);
      photos = photos.filter((photo) => photo.projectId !== id);
    },
    async setStep(projectId, step, marked) {
      projects = projects.map((project) => project.id === projectId ? { ...project, steps: project.steps.map((item) => item.id === step ? { ...item, marked } : item) } : project);
    },
    async listPhotos(projectId) { return photos.filter((photo) => photo.projectId === projectId).map(clonePhoto); },
    async addPhoto(photo) { photos = [...photos, clonePhoto(photo)]; },
    async deletePhoto(id) { photos = photos.filter((photo) => photo.id !== id); },
    async saveQuality(id, quality) { photos = photos.map((photo) => photo.id === id ? { ...photo, quality } : photo); },
    async dump() { return { projects: projects.map(cloneProject), photos: photos.map(clonePhoto) }; },
    async load(snapshot) { projects = snapshot.projects.map(cloneProject); photos = snapshot.photos.map(clonePhoto); },
  };
}

export function idbStore(): CaptureStore {
  const memory = memoryStore();
  return {
    listProjects: () => withDb((db) => request(db.transaction("projects").objectStore("projects").getAll())),
    createProject: async (name) => {
      const project = await memory.createProject(name);
      await withDb((db) => write(db, "projects", project));
      return project;
    },
    deleteProject: async (id) => {
      await withDb(async (db) => {
        const photos = await request<CapturePhoto[]>(db.transaction("photos").objectStore("photos").index("projectId").getAll(id));
        const tx = db.transaction(["projects", "photos"], "readwrite");
        tx.objectStore("projects").delete(id);
        for (const photo of photos) tx.objectStore("photos").delete(photo.id);
        await transactionDone(tx);
      });
    },
    setStep: async (projectId, step, marked) => {
      await withDb(async (db) => {
        const project = await request<CaptureProject | undefined>(db.transaction("projects").objectStore("projects").get(projectId));
        if (!project) return;
        project.steps = project.steps.map((item) => item.id === step ? { ...item, marked } : item);
        await write(db, "projects", project);
      });
    },
    listPhotos: (projectId) => withDb((db) => request(db.transaction("photos").objectStore("photos").index("projectId").getAll(projectId))),
    addPhoto: (photo) => withDb((db) => write(db, "photos", photo)),
    deletePhoto: (id) => withDb((db) => deleteKey(db, "photos", id)),
    saveQuality: async (id, quality) => {
      await withDb(async (db) => {
        const photo = await request<CapturePhoto | undefined>(db.transaction("photos").objectStore("photos").get(id));
        if (!photo) return;
        photo.quality = quality;
        await write(db, "photos", photo);
      });
    },
    dump: async () => ({
      projects: await withDb((db) => request(db.transaction("projects").objectStore("projects").getAll())),
      photos: await withDb((db) => request(db.transaction("photos").objectStore("photos").getAll())),
    }),
    load: async (snapshot) => {
      await withDb(async (db) => {
        const tx = db.transaction(["projects", "photos"], "readwrite");
        tx.objectStore("projects").clear();
        tx.objectStore("photos").clear();
        for (const project of snapshot.projects) tx.objectStore("projects").put(project);
        for (const photo of snapshot.photos) tx.objectStore("photos").put(photo);
        await transactionDone(tx);
      });
    },
  };
}

function cloneProject(project: CaptureProject): CaptureProject {
  return { ...project, steps: project.steps.map((step) => ({ ...step })) };
}

function clonePhoto(photo: CapturePhoto): CapturePhoto {
  return { ...photo, quality: photo.quality ? { ...photo.quality } : null };
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("ooosplat-capture-v1", 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("projects")) db.createObjectStore("projects", { keyPath: "id" });
      if (!db.objectStoreNames.contains("photos")) {
        const photos = db.createObjectStore("photos", { keyPath: "id" });
        photos.createIndex("projectId", "projectId");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withDb<T>(work: (db: IDBDatabase) => Promise<T> | T): Promise<T> {
  const db = await openDb();
  try {
    return await work(db);
  } finally {
    db.close();
  }
}

function request<T>(query: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    query.onsuccess = () => resolve(query.result);
    query.onerror = () => reject(mapStoreError(query.error));
  });
}

function write(db: IDBDatabase, store: "projects" | "photos", value: unknown): Promise<void> {
  const tx = db.transaction(store, "readwrite");
  tx.objectStore(store).put(value);
  return transactionDone(tx);
}

function deleteKey(db: IDBDatabase, store: "photos", id: string): Promise<void> {
  const tx = db.transaction(store, "readwrite");
  tx.objectStore(store).delete(id);
  return transactionDone(tx);
}

function transactionDone(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(mapStoreError(tx.error));
    tx.onabort = () => reject(mapStoreError(tx.error));
  });
}

function mapStoreError(error: DOMException | null): Error {
  if (error?.name === "QuotaExceededError") return new Error("存储空间不足，请先删除一些照片。");
  return error ?? new Error("本地存储失败");
}
