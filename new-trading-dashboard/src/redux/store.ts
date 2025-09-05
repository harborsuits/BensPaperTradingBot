import { configureStore } from '@reduxjs/toolkit';
import globalFiltersReducer from './slices/globalFiltersSlice';

export const store = configureStore({
  reducer: {
    globalFilters: globalFiltersReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
