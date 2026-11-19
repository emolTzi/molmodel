import { describe, expect, it } from "vitest";
import { memoryStore } from "./store";

describe("capture project recovery", () => {
  it("restores projects and photos from a snapshot and can delete them", async () => {
    const store = memoryStore();
    const project = await store.createProject("茶壶");
    await store.addPhoto({
      id: "22222222-2222-4222-8222-222222222222",
      projectId: project.id,
      createdAt: "2026-11-19T00:00:00.000Z",
      source: "file",
      width: 8,
      height: 8,
      blob: new Blob(["jpeg"], { type: "image/jpeg" }),
      quality: null,
    });
    const restored = memoryStore();
    await restored.load(await store.dump());
    expect((await restored.listProjects()).map((item) => item.name)).toEqual(["茶壶"]);
    expect(await restored.listPhotos(project.id)).toHaveLength(1);
    await restored.deletePhoto("22222222-2222-4222-8222-222222222222");
    await restored.deleteProject(project.id);
    expect(await restored.listProjects()).toEqual([]);
    expect(await restored.listPhotos(project.id)).toEqual([]);
  });
});
