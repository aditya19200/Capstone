function LegalTextInput({ value, onChange, placeholder }) {
  return (
    <div className="dashboard-card">
      <label
        htmlFor="legal-text"
        className="section-kicker mb-3 block"
      >
        Legal Document Text
      </label>
      <textarea
        id="legal-text"
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className="min-h-[280px] w-full leading-7"
      />
    </div>
  )
}

export default LegalTextInput
