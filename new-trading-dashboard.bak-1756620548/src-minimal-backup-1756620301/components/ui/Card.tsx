import React from "react";

export function Card({ children, className="" }: React.PropsWithChildren<{className?:string}>) {
  return <section className={`card ${className}`}>{children}</section>;
}

export function CardHeader({ title, right }: { title: React.ReactNode; right?: React.ReactNode }) {
  return (
    <header className="card-header">
      <h3 className="card-title">{title}</h3>
      {right}
    </header>
  );
}

export function CardTitle({ children, className="" }: React.PropsWithChildren<{className?:string}>) {
  return <h3 className={`card-title ${className}`}>{children}</h3>;
}

export function CardDescription({ children, className="" }: React.PropsWithChildren<{className?:string}>) {
  return <p className={`text-sm text-gray-500 ${className}`}>{children}</p>;
}

export function CardContent({ children, className="" }: React.PropsWithChildren<{className?:string}>) {
  return <div className={`card-content ${className}`}>{children}</div>;
}

export function CardFooter({ children }: React.PropsWithChildren) {
  return <footer className="px-4 pb-4">{children}</footer>;
}