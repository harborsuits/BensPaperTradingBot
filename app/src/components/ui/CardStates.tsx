export const CardSkeleton = ({ lines=4 }:{lines?:number}) => (
  <div className="space-y-2">{Array.from({length:lines}).map((_,i)=>
    <div key={i} className="skeleton h-4" />)}</div>
);

export const CardEmpty = ({ text="No data" }:{text?:string}) =>
  <div className="text-sm text-gray-500">{text}</div>;

export const CardError = ({ message }:{message:string}) =>
  <div className="text-sm text-red-600">Error: {message}</div>;


