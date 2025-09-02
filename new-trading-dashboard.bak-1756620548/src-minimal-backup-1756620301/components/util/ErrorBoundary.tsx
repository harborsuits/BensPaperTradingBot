import { Component, type ReactNode } from 'react';

type State = { hasError: boolean };

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(err: unknown) {
    // eslint-disable-next-line no-console
    console.error('UI error:', err);
  }

  render() {
    if (this.state.hasError) {
      return <div className="text-sm text-red-400">Something went wrong rendering this section.</div>;
    }
    return this.props.children;
  }
}

export default ErrorBoundary;


