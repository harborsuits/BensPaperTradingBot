export default function ErrorState({ text='Temporarily unavailable' }:{text?:string}) {
  return (
    <div style={{
      marginTop:8, padding:10, borderRadius:8,
      border:'1px solid #fecaca', background:'rgba(254,226,226,.4)', color:'#991b1b', fontSize:13
    }}>
      {text}
    </div>
  );
}


