export class TileProvider {
  constructor(id, envVar, displayName) {
    if (this.constructor === TileProvider) {
      throw new Error('TileProvider is abstract — use a subclass');
    }
    this.id = id;
    this.envVar = envVar;
    this.displayName = displayName ?? id;
  }

  async validateKey(_key) {
    throw new Error(`Provider '${this.id}' not implement validateKey()`);
  }

  getCesiumToken(_key) {
    throw new Error(`Provider '${this.id}' not implement getCesiumToken()`);
  }

  buildTilesetJs(_token, _cfg) {
    throw new Error(`Provider '${this.id}' not implement buildTilesetJs()`);
  }

  getTilesetWaitMs() {
    return 30_000;
  }
}
