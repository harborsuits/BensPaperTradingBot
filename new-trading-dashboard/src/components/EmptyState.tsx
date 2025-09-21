export default function EmptyState({ text='No data' }:{text?:string}) {
  return <div style={{padding:12, fontSize:13, color:'#64748b'}}>{text}</div>;
}


