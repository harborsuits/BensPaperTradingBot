/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_RQ_DEVTOOLS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}


